ALTER TABLE [dbo].[Administrator] ADD CONSTRAINT [DF_Administrator_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Administrator] ADD CONSTRAINT [DF_Administrator_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Administrator] ADD CONSTRAINT [DF_Administrator_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
