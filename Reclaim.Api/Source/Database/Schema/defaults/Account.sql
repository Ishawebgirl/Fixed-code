ALTER TABLE [dbo].[Account] ADD CONSTRAINT [DF_Account_NiceName] DEFAULT (N'X') FOR [NiceName]
GO
ALTER TABLE [dbo].[Account] ADD CONSTRAINT [DF_Account_Guid] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Account] ADD CONSTRAINT [DF_Account_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Account] ADD CONSTRAINT [DF_Account_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
