ALTER TABLE [dbo].[ApplicationSetting] ADD CONSTRAINT [DF_ApplicationSetting_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ApplicationSetting] ADD CONSTRAINT [DF_ApplicationSetting_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
