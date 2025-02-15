ALTER TABLE [dbo].[Document] ADD CONSTRAINT [DF_Document_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Document] ADD CONSTRAINT [DF_Document_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Document] ADD CONSTRAINT [DF_Document_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
