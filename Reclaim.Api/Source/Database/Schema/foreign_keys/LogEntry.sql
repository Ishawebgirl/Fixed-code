ALTER TABLE [dbo].[LogEntry] WITH CHECK ADD CONSTRAINT [FK_LogEntry_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[LogEntry] WITH CHECK ADD CONSTRAINT [FK_LogEntry_ErrorCode]
   FOREIGN KEY([ErrorCodeID]) REFERENCES [dbo].[ErrorCode] ([ErrorCodeID])

GO
ALTER TABLE [dbo].[LogEntry] WITH CHECK ADD CONSTRAINT [FK_LogEntry_LogEntryLevel]
   FOREIGN KEY([LogEntryLevelID]) REFERENCES [dbo].[LogEntryLevel] ([LogEntryLevelID])

GO
