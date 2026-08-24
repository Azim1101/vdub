package com.azim.vdub

import com.azim.vdub.net.VideoResolver
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class VideoResolverTest {

    private val r = VideoResolver()

    @Test
    fun `finds video tag src and makes it absolute`() {
        assertEquals(
            "https://site.com/media/ep1.mp4",
            r.extractMediaUrl("""<video controls src="/media/ep1.mp4"></video>""",
                "https://site.com/watch/1")
        )
    }

    @Test
    fun `finds source tag`() {
        assertEquals(
            "https://cdn.x.com/a.webm",
            r.extractMediaUrl(
                """<video><source src="https://cdn.x.com/a.webm" type="video/webm"></video>""",
                "https://x.com/"
            )
        )
    }

    @Test
    fun `finds og video and resolves protocol relative`() {
        assertEquals(
            "https://cdn.z.com/s.mp4",
            r.extractMediaUrl(
                """<meta property="og:video:secure_url" content="//cdn.z.com/s.mp4">""",
                "https://z.com/"
            )
        )
    }

    @Test
    fun `finds json contentUrl with escaped slashes`() {
        assertEquals(
            "https://cdn.a.com/f.mp4",
            r.extractMediaUrl("""{"contentUrl":"https:\/\/cdn.a.com\/f.mp4"}""", "https://a.com/")
        )
    }

    @Test
    fun `finds hls playlist`() {
        assertEquals(
            "https://cdn.b.com/master.m3u8",
            r.extractMediaUrl("""{"hlsUrl":"https://cdn.b.com/master.m3u8"}""", "https://b.com/")
        )
    }

    @Test
    fun `keeps query string and decodes entities`() {
        assertEquals(
            "https://cdn.d.com/y.mp4?a=1&b=2",
            r.extractMediaUrl("""<video src="https://cdn.d.com/y.mp4?a=1&amp;b=2">""",
                "https://d.com/")
        )
    }

    @Test
    fun `resolves relative path against directory`() {
        assertEquals(
            "https://e.com/show/clips/ep.mp4",
            r.extractMediaUrl("""<source src="clips/ep.mp4">""", "https://e.com/show/")
        )
    }

    @Test
    fun `returns null when the page has no media`() {
        assertNull(r.extractMediaUrl("""<div id="player" data-cfg="x"></div>""", "https://f.com/"))
        assertNull(r.extractMediaUrl("""<video poster="/img/p.jpg"></video>""", "https://g.com/"))
    }
}
