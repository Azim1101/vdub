package com.azim.vdub.data.local

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.azim.vdub.core.ModelCatalog
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

private val Context.voiceDataStore by preferencesDataStore(name = "vdub_voice")

/**
 * Which voice engine the user picked.
 *
 * Persisted rather than held in memory: the voice stage can take hours, and a
 * process death mid-run must not silently switch engines and produce clips in
 * two different voices.
 */
@Singleton
class VoicePrefs @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val key = stringPreferencesKey("voice_engine_id")
    private val lastProjectKey = stringPreferencesKey("last_project")

    /** Defaults to the Q4 pack — the smaller download. */
    val engineId: Flow<String> = context.voiceDataStore.data.map { prefs ->
        val stored = prefs[key]
        // Guard against an id left over from a renamed/removed model.
        if (stored != null && ModelCatalog.VOICE_ENGINES.any { it.id == stored }) stored
        else ModelCatalog.VOICE_ENGINES.first().id
    }

    suspend fun setEngine(id: String) {
        require(ModelCatalog.VOICE_ENGINES.any { it.id == id }) { "Unknown voice engine: $id" }
        context.voiceDataStore.edit { it[key] = id }
    }

    /**
     * The project the user last worked on, so relaunching reopens it instead
     * of dropping them on an empty Step 1.
     */
    val lastProject: Flow<String?> = context.voiceDataStore.data.map { it[lastProjectKey] }

    suspend fun setLastProject(name: String) {
        if (name.isBlank()) return
        context.voiceDataStore.edit { it[lastProjectKey] = name }
    }
}
