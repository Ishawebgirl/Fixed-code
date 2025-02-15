ALTER TABLE [dbo].[IdentityProvider] ADD CONSTRAINT [DF_IdentityProvider_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[IdentityProvider] ADD CONSTRAINT [DF_IdentityProvider_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
