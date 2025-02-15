ALTER TABLE [dbo].[ChatType] ADD CONSTRAINT [DF_ChatType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ChatType] ADD CONSTRAINT [DF_ChatType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
