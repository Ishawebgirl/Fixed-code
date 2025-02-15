ALTER TABLE [dbo].[RoofType] ADD CONSTRAINT [DF_RoofType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[RoofType] ADD CONSTRAINT [DF_RoofType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
