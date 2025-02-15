ALTER TABLE [dbo].[Job] ADD CONSTRAINT [DF_Job_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Job] ADD CONSTRAINT [DF_Job_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
