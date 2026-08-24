package com.azim.vdub.audio

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal RIFF/WAVE reader+writer for 16-bit PCM.
 * Deliberately dependency-free: the pipeline treats org_audio.wav as a raw
 * sample array (numpy-style slicing), so we never shell out to ffmpeg.
 */
object WavIo {

    const val HEADER_BYTES = 44

    data class Format(
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int,
        val dataOffset: Long,
        val dataBytes: Long
    ) {
        val frameCount: Long get() = dataBytes / (channels * bitsPerSample / 8)
        val durationSec: Double get() = frameCount.toDouble() / sampleRate
    }

    fun writeHeader(raf: RandomAccessFile, sampleRate: Int, channels: Int, dataBytes: Int) {
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8
        val buf = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray())
        buf.putInt(36 + dataBytes)
        buf.put("WAVE".toByteArray())
        buf.put("fmt ".toByteArray())
        buf.putInt(16)                       // PCM chunk size
        buf.putShort(1)                      // format = PCM
        buf.putShort(channels.toShort())
        buf.putInt(sampleRate)
        buf.putInt(byteRate)
        buf.putShort(blockAlign.toShort())
        buf.putShort(bitsPerSample.toShort())
        buf.put("data".toByteArray())
        buf.putInt(dataBytes)
        raf.seek(0)
        raf.write(buf.array())
    }

    /** Write a complete 16-bit PCM wav in one shot. */
    fun writePcm16(target: File, pcm: ByteArray, sampleRate: Int, channels: Int = 1) {
        target.parentFile?.mkdirs()
        RandomAccessFile(target, "rw").use { raf ->
            raf.setLength(0)
            writeHeader(raf, sampleRate, channels, pcm.size)
            raf.seek(HEADER_BYTES.toLong())
            raf.write(pcm)
        }
    }

    /** Parse the header, tolerating extra chunks (LIST/fact) before `data`. */
    fun readFormat(file: File): Format {
        RandomAccessFile(file, "r").use { raf ->
            val riff = ByteArray(12)
            raf.readFully(riff)
            require(String(riff, 0, 4) == "RIFF" && String(riff, 8, 4) == "WAVE") {
                "Not a RIFF/WAVE file: ${file.name}"
            }
            var sampleRate = 16_000
            var channels = 1
            var bits = 16
            val hdr = ByteArray(8)
            // <= so a chunk header ending exactly at EOF is still read; an
            // empty `data` chunk (44-byte wav, zero-length clip) sits there.
            while (raf.filePointer <= raf.length() - 8) {
                raf.readFully(hdr)
                val id = String(hdr, 0, 4)
                val size = ByteBuffer.wrap(hdr, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong()
                when (id) {
                    "fmt " -> {
                        val fmt = ByteArray(size.toInt().coerceAtLeast(16))
                        raf.readFully(fmt, 0, size.toInt())
                        val bb = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
                        bb.short                       // audioFormat
                        channels = bb.short.toInt()
                        sampleRate = bb.int
                        bb.int                         // byteRate
                        bb.short                       // blockAlign
                        bits = bb.short.toInt()
                    }
                    "data" -> return Format(sampleRate, channels, bits, raf.filePointer, size)
                    else -> raf.seek(raf.filePointer + size + (size and 1L))
                }
            }
            error("No data chunk in ${file.name}")
        }
    }

    /**
     * Slice [startSec, endSec) out of a wav and save as a new wav — the
     * numpy-slicing replacement for `ffmpeg -ss -to`. Reads only the needed
     * byte range, so cutting 190 clips out of a 70 MB file stays cheap.
     */
    fun sliceToFile(
        source: File,
        fmt: Format,
        startSec: Double,
        endSec: Double,
        target: File
    ): Int {
        val bytesPerFrame = fmt.channels * fmt.bitsPerSample / 8
        val startFrame = (startSec * fmt.sampleRate).toLong().coerceIn(0, fmt.frameCount)
        val endFrame = (endSec * fmt.sampleRate).toLong().coerceIn(startFrame, fmt.frameCount)
        val frames = (endFrame - startFrame).toInt()
        if (frames <= 0) {
            writePcm16(target, ByteArray(0), fmt.sampleRate, fmt.channels)
            return 0
        }
        val byteLen = frames * bytesPerFrame
        val pcm = ByteArray(byteLen)
        RandomAccessFile(source, "r").use { raf ->
            raf.seek(fmt.dataOffset + startFrame * bytesPerFrame)
            raf.readFully(pcm)
        }
        writePcm16(target, pcm, fmt.sampleRate, fmt.channels)
        return frames
    }
}
