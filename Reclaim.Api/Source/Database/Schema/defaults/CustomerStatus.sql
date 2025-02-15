ALTER TABLE [dbo].[CustomerStatus] ADD CONSTRAINT [DF_CustomerStatus_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[CustomerStatus] ADD CONSTRAINT [DF_CustomerStatus_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
