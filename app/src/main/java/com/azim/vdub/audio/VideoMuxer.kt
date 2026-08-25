package com.azim.vdub.audio

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import kotlin.coroutines.coroutineContext

/**
 * Writes the dubbed audio back onto the original video.
 *
 * The video track is **copied, not re-encoded** — samples are passed straight
 * from extractor to muxer. A 142 MB file therefore muxes in seconds with no
 * quality loss, where a re-encode would take many minutes and degrade it.
 *
 * Only the audio is encoded (AAC), because it is new.
 */
object VideoMuxer {

    private const val TIMEOUT_US = 10_000L
    private const val AAC_BITRATE = 128_000

    /**
     * @param dubbed 24 kHz mono float samples in [-1, 1]
     * @return the written file
     */
    suspend fun mux(
        sourceVideo: File,
        dubbed: FloatArray,
        sampleRate: Int,
        target: File,
        onProgress: (Float) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {
        require(sourceVideo.exists()) { "video missing: ${sourceVideo.name}" }
        require(dubbed.isNotEmpty()) { "no dubbed audio to write" }
        target.parentFile?.mkdirs()
        if (target.exists()) target.delete()

        val extractor = MediaExtractor()
        extractor.setDataSource(sourceVideo.absolutePath)

        val videoTrack = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                ?.startsWith("video/") == true
        } ?: run {
            extractor.release()
            error("No video track in ${sourceVideo.name}")
        }

        val videoFormat = extractor.getTrackFormat(videoTrack)
        val durationUs = runCatching { videoFormat.getLong(MediaFormat.KEY_DURATION) }
            .getOrDefault(0L)

        val muxer = MediaMuxer(target.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        val outVideoTrack = muxer.addTrack(videoFormat)

        // Encode the dubbed audio first so its format is known before start().
        val encoder = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
        val audioFormat = MediaFormat.createAudioFormat(
            MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, 1
        ).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, AAC_BITRATE)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16384)
        }
        encoder.configure(audioFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder.start()

        val pcm = toPcm16(dubbed)
        var outAudioTrack = -1
        var started = false
        var inputOffset = 0
        var sawInputEos = false
        var sawOutputEos = false
        val info = MediaCodec.BufferInfo()
        val pendingAudio = ArrayList<Pair<ByteArray, MediaCodec.BufferInfo>>()

        try {
            // ---- encode audio ----
            while (!sawOutputEos) {
                coroutineContext.ensureActive()

                if (!sawInputEos) {
                    val inIndex = encoder.dequeueInputBuffer(TIMEOUT_US)
                    if (inIndex >= 0) {
                        val buf = encoder.getInputBuffer(inIndex)!!
                        buf.clear()
                        val chunk = minOf(buf.capacity(), pcm.size - inputOffset)
                        if (chunk <= 0) {
                            encoder.queueInputBuffer(
                                inIndex, 0, 0, ptsFor(inputOffset, sampleRate),
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            sawInputEos = true
                        } else {
                            buf.put(pcm, inputOffset, chunk)
                            encoder.queueInputBuffer(
                                inIndex, 0, chunk, ptsFor(inputOffset, sampleRate), 0
                            )
                            inputOffset += chunk
                            onProgress(0.5f * inputOffset / pcm.size)
                        }
                    }
                }

                when (val outIndex = encoder.dequeueOutputBuffer(info, TIMEOUT_US)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        outAudioTrack = muxer.addTrack(encoder.outputFormat)
                        muxer.start()
                        started = true
                        // Anything encoded before start() has to wait for it.
                        pendingAudio.forEach { (bytes, bi) ->
                            muxer.writeSampleData(outAudioTrack, ByteBuffer.wrap(bytes), bi)
                        }
                        pendingAudio.clear()
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outIndex >= 0) {
                        val outBuf = encoder.getOutputBuffer(outIndex)!!
                        val isConfig =
                            info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0
                        if (info.size > 0 && !isConfig) {
                            val copy = ByteArray(info.size)
                            outBuf.position(info.offset)
                            outBuf.get(copy)
                            val bi = MediaCodec.BufferInfo().apply {
                                set(0, copy.size, info.presentationTimeUs, info.flags)
                            }
                            if (started && outAudioTrack >= 0) {
                                muxer.writeSampleData(
                                    outAudioTrack, ByteBuffer.wrap(copy), bi
                                )
                            } else {
                                pendingAudio.add(copy to bi)
                            }
                        }
                        encoder.releaseOutputBuffer(outIndex, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawOutputEos = true
                        }
                    }
                }
            }

            check(started) { "audio encoder produced no output format" }

            // ---- copy the video track verbatim ----
            extractor.selectTrack(videoTrack)
            val buffer = ByteBuffer.allocate(1 shl 20)
            val videoInfo = MediaCodec.BufferInfo()
            while (true) {
                coroutineContext.ensureActive()
                val size = extractor.readSampleData(buffer, 0)
                if (size < 0) break
                videoInfo.offset = 0
                videoInfo.size = size
                videoInfo.presentationTimeUs = extractor.sampleTime
                videoInfo.flags = extractor.sampleFlags
                muxer.writeSampleData(outVideoTrack, buffer, videoInfo)
                if (durationUs > 0) {
                    onProgress(0.5f + 0.5f * (extractor.sampleTime.toFloat() / durationUs))
                }
                extractor.advance()
            }
        } finally {
            runCatching { encoder.stop() }
            runCatching { encoder.release() }
            runCatching { muxer.stop() }
            runCatching { muxer.release() }
            runCatching { extractor.release() }
        }

        onProgress(1f)
        target
    }

    private fun ptsFor(byteOffset: Int, sampleRate: Int): Long =
        (byteOffset / 2L) * 1_000_000L / sampleRate

    private fun toPcm16(samples: FloatArray): ByteArray {
        val out = ByteArray(samples.size * 2)
        var j = 0
        for (s in samples) {
            val v = (s.coerceIn(-1f, 1f) * 32767f).toInt()
            out[j++] = (v and 0xFF).toByte()
            out[j++] = ((v shr 8) and 0xFF).toByte()
        }
        return out
    }
}
