ALTER TABLE [dbo].[AccountAuthentication] ADD CONSTRAINT [DF_AccountAuthentication_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[AccountAuthentication] ADD CONSTRAINT [DF_AccountAuthentication_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
