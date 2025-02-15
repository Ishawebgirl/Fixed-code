ALTER TABLE [dbo].[JobEvent] WITH CHECK ADD CONSTRAINT [FK_JobEvent_Job]
   FOREIGN KEY([JobID]) REFERENCES [dbo].[Job] ([JobID])

GO
ALTER TABLE [dbo].[JobEvent] WITH CHECK ADD CONSTRAINT [FK_JobEvent_LogEntry]
   FOREIGN KEY([LogEntryID]) REFERENCES [dbo].[LogEntry] ([LogEntryID])

GO
