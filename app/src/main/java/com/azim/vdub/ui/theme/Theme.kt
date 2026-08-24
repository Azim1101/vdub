package com.azim.vdub.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

private val Violet = Color(0xFF7C5CFF)
private val VioletDim = Color(0xFF4A3AA8)
private val Aqua = Color(0xFF35D6C4)
private val Amber = Color(0xFFFFB020)
private val Ink = Color(0xFF12131A)
private val InkCard = Color(0xFF1B1D27)

private val DarkColors = darkColorScheme(
    primary = Violet,
    onPrimary = Color.White,
    primaryContainer = VioletDim,
    onPrimaryContainer = Color(0xFFE8E3FF),
    secondary = Aqua,
    onSecondary = Color(0xFF00261F),
    tertiary = Amber,
    onTertiary = Color(0xFF2B1A00),
    background = Ink,
    onBackground = Color(0xFFE6E7EE),
    surface = Ink,
    onSurface = Color(0xFFE6E7EE),
    surfaceVariant = InkCard,
    onSurfaceVariant = Color(0xFFB9BCCB),
    outline = Color(0xFF3A3D4D),
    error = Color(0xFFFF6B6B)
)

private val LightColors = lightColorScheme(
    primary = Violet,
    secondary = Color(0xFF00A896),
    tertiary = Color(0xFFB07000)
)

private val VdubTypography = Typography(
    titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
    bodyMedium = TextStyle(fontSize = 14.sp),
    bodySmall = TextStyle(fontSize = 12.sp),
    labelLarge = TextStyle(fontSize = 14.sp, fontWeight = FontWeight.Medium)
)

@Composable
fun VdubTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = VdubTypography,
        content = content
    )
}
