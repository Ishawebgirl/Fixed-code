ALTER TABLE [dbo].[Chat] ADD CONSTRAINT [DF_Chat_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Chat] ADD CONSTRAINT [DF_Chat_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Chat] ADD CONSTRAINT [DF_Chat_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
