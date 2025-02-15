ALTER TABLE [dbo].[Role] ADD CONSTRAINT [DF_Role_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Role] ADD CONSTRAINT [DF_Role_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
