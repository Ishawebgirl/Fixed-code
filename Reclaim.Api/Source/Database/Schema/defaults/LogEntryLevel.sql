ALTER TABLE [dbo].[LogEntryLevel] ADD CONSTRAINT [DF_LogEntryLevel_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[LogEntryLevel] ADD CONSTRAINT [DF_LogEntryLevel_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
