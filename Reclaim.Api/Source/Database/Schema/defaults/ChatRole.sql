ALTER TABLE [dbo].[ChatRole] ADD CONSTRAINT [DF_ChatRole_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ChatRole] ADD CONSTRAINT [DF_ChatRole_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
