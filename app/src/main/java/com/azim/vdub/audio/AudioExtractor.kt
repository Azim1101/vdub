package com.azim.vdub.audio

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.coroutines.coroutineContext

/**
 * Video -> 16 kHz mono PCM16 wav, using the platform MediaCodec decoder.
 *
 * Why not FFmpegKit: the spec hit a repeatable segfault extracting audio with
 * ffmpeg/FFmpegKit on this device class. MediaCodec is hardware-backed, has no
 * native .so of our own to crash, and streams straight to disk so a 2267 s /
 * 70 MB track never lands in RAM.
 */
object AudioExtractor {

    const val TARGET_SAMPLE_RATE = 16_000
    private const val TIMEOUT_US = 10_000L

    /**
     * @param onProgress 0f..1f based on presentation timestamps.
     * @return the written wav file.
     */
    suspend fun extractToWav16kMono(
        video: File,
        target: File,
        onProgress: (Float) -> Unit = {}
    ): File = withContext(Dispatchers.IO) {
        require(video.exists()) { "Video not found: ${video.absolutePath}" }
        target.parentFile?.mkdirs()

        val extractor = MediaExtractor()
        extractor.setDataSource(video.absolutePath)

        val trackIndex = (0 until extractor.trackCount).firstOrNull { i ->
            extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME)
                ?.startsWith("audio/") == true
        } ?: run {
            extractor.release()
            error("No audio track in ${video.name}")
        }

        val inputFormat = extractor.getTrackFormat(trackIndex)
        extractor.selectTrack(trackIndex)

        val mime = inputFormat.getString(MediaFormat.KEY_MIME)!!
        val srcRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val srcChannels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val durationUs = runCatching { inputFormat.getLong(MediaFormat.KEY_DURATION) }
            .getOrDefault(0L)

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(inputFormat, null, null, 0)
        codec.start()

        val raf = RandomAccessFile(target, "rw")
        raf.setLength(0)
        raf.seek(WavIo.HEADER_BYTES.toLong())   // header patched at the end

        var totalBytes = 0
        var outRate = srcRate
        var outChannels = srcChannels
        val resampler = LinearResampler()
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEos = false
        var sawOutputEos = false
        var lastReported = -1

        try {
            while (!sawOutputEos) {
                coroutineContext.ensureActive()

                if (!sawInputEos) {
                    val inIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                    if (inIndex >= 0) {
                        val inBuf = codec.getInputBuffer(inIndex)!!
                        val sampleSize = extractor.readSampleData(inBuf, 0)
                        if (sampleSize < 0) {
                            codec.queueInputBuffer(
                                inIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            sawInputEos = true
                        } else {
                            codec.queueInputBuffer(
                                inIndex, 0, sampleSize, extractor.sampleTime, 0
                            )
                            extractor.advance()
                        }
                    }
                }

                when (val outIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val f = codec.outputFormat
                        outRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        outChannels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                    else -> if (outIndex >= 0) {
                        val outBuf = codec.getOutputBuffer(outIndex)!!
                        if (bufferInfo.size > 0) {
                            outBuf.position(bufferInfo.offset)
                            outBuf.limit(bufferInfo.offset + bufferInfo.size)
                            val mono = toMono16(outBuf, outChannels)
                            val resampled = resampler.process(mono, outRate, TARGET_SAMPLE_RATE)
                            val bytes = shortsToLe(resampled)
                            raf.write(bytes)
                            totalBytes += bytes.size
                        }
                        codec.releaseOutputBuffer(outIndex, false)
                        if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            sawOutputEos = true
                        }
                        if (durationUs > 0) {
                            val pct = ((bufferInfo.presentationTimeUs.toFloat() /
                                durationUs) * 100).toInt().coerceIn(0, 100)
                            if (pct != lastReported) {
                                lastReported = pct
                                onProgress(pct / 100f)
                            }
                        }
                    }
                }
            }
            WavIo.writeHeader(raf, TARGET_SAMPLE_RATE, 1, totalBytes)
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
            runCatching { extractor.release() }
            runCatching { raf.close() }
        }
        onProgress(1f)
        target
    }

    /** Downmix interleaved PCM16 to mono shorts. */
    private fun toMono16(buf: ByteBuffer, channels: Int): ShortArray {
        val sb = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val total = sb.remaining()
        if (channels <= 1) {
            val out = ShortArray(total)
            sb.get(out)
            return out
        }
        val frames = total / channels
        val out = ShortArray(frames)
        val tmp = ShortArray(total)
        sb.get(tmp)
        for (i in 0 until frames) {
            var acc = 0
            for (c in 0 until channels) acc += tmp[i * channels + c]
            out[i] = (acc / channels).toShort()
        }
        return out
    }

    private fun shortsToLe(samples: ShortArray): ByteArray {
        val out = ByteArray(samples.size * 2)
        var j = 0
        for (s in samples) {
            out[j++] = (s.toInt() and 0xFF).toByte()
            out[j++] = ((s.toInt() shr 8) and 0xFF).toByte()
        }
        return out
    }

    /**
     * Stateful linear resampler — keeps the fractional cursor and the last
     * sample across decoder buffers so there are no clicks at buffer joins.
     */
    private class LinearResampler {
        private var pos = 0.0
        private var carry: Short? = null

        fun process(input: ShortArray, inRate: Int, outRate: Int): ShortArray {
            if (input.isEmpty()) return input
            if (inRate == outRate) return input
            val step = inRate.toDouble() / outRate
            val prev = carry
            val src = if (prev != null) ShortArray(input.size + 1).also {
                it[0] = prev; input.copyInto(it, 1)
            } else input

            val out = ArrayList<Short>((src.size / step).toInt() + 2)
            var p = pos
            while (p < src.size - 1) {
                val i = p.toInt()
                val frac = p - i
                val v = src[i] + (src[i + 1] - src[i]) * frac
                out.add(v.toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
                    .toShort())
                p += step
            }
            pos = p - (src.size - 1)
            carry = src.last()
            return out.toShortArray()
        }
    }
}
