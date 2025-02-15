ALTER TABLE [dbo].[JobStatus] ADD CONSTRAINT [DF_JobStatus_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[JobStatus] ADD CONSTRAINT [DF_JobStatus_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
