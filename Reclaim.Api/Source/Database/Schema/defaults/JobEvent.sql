ALTER TABLE [dbo].[JobEvent] ADD CONSTRAINT [DF_JobEvent_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[JobEvent] ADD CONSTRAINT [DF_JobEvent_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
