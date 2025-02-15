ALTER TABLE [dbo].[LogEntry] ADD CONSTRAINT [DF_LogEntry_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[LogEntry] ADD CONSTRAINT [DF_LogEntry_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
