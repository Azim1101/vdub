package com.azim.vdub.data.local

import androidx.room.ColumnInfo
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import androidx.room.Upsert
import kotlinx.coroutines.flow.Flow

/**
 * Resume-safe state. Mirrors the S0x.done markers on disk so the UI can
 * reopen a project and jump straight to the first unfinished step.
 */
@Entity(tableName = "projects")
data class ProjectEntity(
    @PrimaryKey val name: String,
    @ColumnInfo(name = "video_path") val videoPath: String? = null,
    @ColumnInfo(name = "video_source") val videoSource: String = "NONE",
    @ColumnInfo(name = "source_url") val sourceUrl: String? = null,
    @ColumnInfo(name = "duration_ms") val durationMs: Long = 0L,
    @ColumnInfo(name = "srt_path") val srtPath: String? = null,
    @ColumnInfo(name = "translated_srt_path") val translatedSrtPath: String? = null,
    @ColumnInfo(name = "cue_count") val cueCount: Int = 0,
    @ColumnInfo(name = "line_count") val lineCount: Int = 0,
    @ColumnInfo(name = "clip_count") val clipCount: Int = 0,
    @ColumnInfo(name = "step1_done") val step1Done: Boolean = false,
    @ColumnInfo(name = "updated_at") val updatedAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "clips")
data class ClipEntity(
    @PrimaryKey val id: String,             // "{project}#{lineId}"
    val project: String,
    @ColumnInfo(name = "line_id") val lineId: Int,
    @ColumnInfo(name = "start_sec") val startSec: Double,
    @ColumnInfo(name = "end_sec") val endSec: Double,
    val text: String,
    @ColumnInfo(name = "wav_path") val wavPath: String,
    val speaker: String? = null,
    val emotion: String? = null
)

@Dao
interface ProjectDao {
    @Query("SELECT * FROM projects ORDER BY updated_at DESC")
    fun observeAll(): Flow<List<ProjectEntity>>

    @Query("SELECT * FROM projects WHERE name = :name LIMIT 1")
    fun observe(name: String): Flow<ProjectEntity?>

    @Query("SELECT * FROM projects WHERE name = :name LIMIT 1")
    suspend fun get(name: String): ProjectEntity?

    @Upsert
    suspend fun upsert(project: ProjectEntity)

    @Query("DELETE FROM projects WHERE name = :name")
    suspend fun delete(name: String)
}

@Dao
interface ClipDao {
    @Query("SELECT * FROM clips WHERE project = :project ORDER BY line_id ASC")
    fun observeForProject(project: String): Flow<List<ClipEntity>>

    @Query("SELECT COUNT(*) FROM clips WHERE project = :project")
    suspend fun countFor(project: String): Int

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAll(clips: List<ClipEntity>)

    @Query("DELETE FROM clips WHERE project = :project")
    suspend fun clear(project: String)
}

@Database(
    entities = [ProjectEntity::class, ClipEntity::class],
    version = 1,
    exportSchema = false
)
abstract class VdubDatabase : RoomDatabase() {
    abstract fun projectDao(): ProjectDao
    abstract fun clipDao(): ClipDao
}
