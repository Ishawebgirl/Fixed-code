ALTER TABLE [dbo].[Job] WITH CHECK ADD CONSTRAINT [FK_Job_JobStatus]
   FOREIGN KEY([JobStatusID]) REFERENCES [dbo].[JobStatus] ([JobStatusID])

GO
ALTER TABLE [dbo].[Job] WITH CHECK ADD CONSTRAINT [FK_Job_JobType]
   FOREIGN KEY([JobTypeID]) REFERENCES [dbo].[JobType] ([JobTypeID])

GO
