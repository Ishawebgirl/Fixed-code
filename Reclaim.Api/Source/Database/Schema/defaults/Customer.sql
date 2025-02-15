ALTER TABLE [dbo].[Customer] ADD CONSTRAINT [DF_Customer_UniqueID] DEFAULT (newid()) FOR [UniqueID]
GO
ALTER TABLE [dbo].[Customer] ADD CONSTRAINT [DF_Customer_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Customer] ADD CONSTRAINT [DF_Customer_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
