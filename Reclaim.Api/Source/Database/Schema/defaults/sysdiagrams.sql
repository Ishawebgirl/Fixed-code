ALTER TABLE [dbo].[sysdiagrams] ADD CONSTRAINT [DF_sysdiagrams_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
