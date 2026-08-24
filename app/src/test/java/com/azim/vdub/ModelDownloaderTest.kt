package com.azim.vdub

import com.azim.vdub.net.ModelDownloader
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelDownloaderTest {

    @Test
    fun `campplus spec targets the file the pipeline loads`() {
        val spec = ModelDownloader.CAMPPLUS
        assertEquals("campplus.onnx", spec.fileName)
        // 28 MB, matching the 3D-Speaker CAM++ export
        assertTrue(spec.approxBytes in 25_000_000..32_000_000)
    }

    @Test
    fun `every mirror is https and points at an onnx file`() {
        ModelDownloader.CAMPPLUS.urls.forEach { url ->
            assertTrue("not https: $url", url.startsWith("https://"))
            assertTrue("not onnx: $url", url.substringBefore('?').endsWith(".onnx"))
        }
    }

    @Test
    fun `has more than one mirror so a single outage is survivable`() {
        assertTrue(ModelDownloader.CAMPPLUS.urls.size >= 2)
        assertEquals(
            ModelDownloader.CAMPPLUS.urls.size,
            ModelDownloader.CAMPPLUS.urls.distinct().size
        )
    }

    @Test
    fun `all registered models are listed`() {
        assertTrue(ModelDownloader.ALL.contains(ModelDownloader.CAMPPLUS))
    }
}
