ALTER TABLE [dbo].[Policy] ADD CONSTRAINT [DF_Policy_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Policy] ADD CONSTRAINT [DF_Policy_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Policy] ADD CONSTRAINT [DF_Policy_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
