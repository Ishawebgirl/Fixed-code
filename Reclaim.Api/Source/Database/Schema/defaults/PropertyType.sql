ALTER TABLE [dbo].[PropertyType] ADD CONSTRAINT [DF_PropertyType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[PropertyType] ADD CONSTRAINT [DF_PropertyType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
