ALTER TABLE [dbo].[ChatMessage] ADD CONSTRAINT [DF_ChatMessage_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[ChatMessage] ADD CONSTRAINT [DF_ChatMessage_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ChatMessage] ADD CONSTRAINT [DF_ChatMessage_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
