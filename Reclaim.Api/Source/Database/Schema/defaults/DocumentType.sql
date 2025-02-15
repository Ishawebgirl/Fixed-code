ALTER TABLE [dbo].[DocumentType] ADD CONSTRAINT [DF_DocumentType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[DocumentType] ADD CONSTRAINT [DF_DocumentType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
