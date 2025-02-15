ALTER TABLE [dbo].[Email] ADD CONSTRAINT [DF_Email_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Email] ADD CONSTRAINT [DF_Email_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Email] ADD CONSTRAINT [DF_Email_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
