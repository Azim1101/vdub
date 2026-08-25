package com.azim.vdub

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The HTML guard rejects error pages saved under a model's name. It must not
 * reject legitimate files that merely start with '<' — SenseVoice's tokens.txt
 * begins "<unk> 0", which a naive startsWith("<") threw away.
 */
class ModelDownloadValidationTest {

    private val htmlRe = Regex(
        """^\s*(<!DOCTYPE\s+html|<html\b|<head\b|<body\b|<\?xml)""",
        RegexOption.IGNORE_CASE
    )

    private fun looksLikeHtml(s: String) = htmlRe.containsMatchIn(s.trim())

    @Test
    fun `sensevoice tokens are not html`() {
        assertFalse(looksLikeHtml("<unk> 0\n<s> 1\n</s> 2\n"))
    }

    @Test
    fun `angle bracket tokens are not html`() {
        assertFalse(looksLikeHtml("<pad>\n<eos>\n<bos>"))
    }

    @Test
    fun `real error pages are caught`() {
        listOf(
            "<!DOCTYPE html>\n<html>",
            "<html lang=\"en\"><head>",
            "<head><title>404</title>",
            "<?xml version=\"1.0\"?><Error>",
            "\n\n  <!doctype HTML>"
        ).forEach { assertTrue("missed: $it", looksLikeHtml(it)) }
    }

    @Test
    fun `json and text pass`() {
        assertFalse(looksLikeHtml("""{"weight": [[1.0]]}"""))
        assertFalse(looksLikeHtml("hello world"))
    }
}
