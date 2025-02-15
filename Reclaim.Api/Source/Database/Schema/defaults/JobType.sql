ALTER TABLE [dbo].[JobType] ADD CONSTRAINT [DF_JobType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[JobType] ADD CONSTRAINT [DF_JobType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
