ALTER TABLE [dbo].[ErrorCode] ADD CONSTRAINT [DF_ErrorCode_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ErrorCode] ADD CONSTRAINT [DF_ErrorCode_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
