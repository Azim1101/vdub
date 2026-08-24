package com.azim.vdub.di

import android.content.Context
import androidx.room.Room
import com.azim.vdub.data.local.ClipDao
import com.azim.vdub.data.local.ProjectDao
import com.azim.vdub.data.local.VdubDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext context: Context): VdubDatabase =
        Room.databaseBuilder(context, VdubDatabase::class.java, "vdub.db")
            .fallbackToDestructiveMigration()
            .build()

    @Provides
    fun provideProjectDao(db: VdubDatabase): ProjectDao = db.projectDao()

    @Provides
    fun provideClipDao(db: VdubDatabase): ClipDao = db.clipDao()
}
