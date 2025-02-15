ALTER TABLE [dbo].[AccountPassword] ADD CONSTRAINT [DF_AccountPassword_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[AccountPassword] ADD CONSTRAINT [DF_AccountPassword_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
