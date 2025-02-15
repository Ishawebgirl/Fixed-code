ALTER TABLE [dbo].[OwnershipType] ADD CONSTRAINT [DF_OwnershipType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[OwnershipType] ADD CONSTRAINT [DF_OwnershipType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
