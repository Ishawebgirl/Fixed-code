ALTER TABLE [dbo].[ClaimType] ADD CONSTRAINT [DF_ClaimType_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[ClaimType] ADD CONSTRAINT [DF_ClaimType_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
