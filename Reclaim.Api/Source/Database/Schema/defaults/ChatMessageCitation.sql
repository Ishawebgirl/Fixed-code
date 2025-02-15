ALTER TABLE [dbo].[ChatMessageCitation] ADD CONSTRAINT [DF_ChatMessageCitation_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[ChatMessageCitation] ADD CONSTRAINT [DF_ChatMessageCitation_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ChatMessageCitation] ADD CONSTRAINT [DF_ChatMessageCitation_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
