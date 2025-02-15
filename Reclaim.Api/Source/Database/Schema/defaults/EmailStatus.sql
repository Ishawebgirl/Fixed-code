ALTER TABLE [dbo].[EmailStatus] ADD CONSTRAINT [DF_EmailStatus_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[EmailStatus] ADD CONSTRAINT [DF_EmailStatus_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
