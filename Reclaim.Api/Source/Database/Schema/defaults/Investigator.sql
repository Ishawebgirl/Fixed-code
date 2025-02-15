ALTER TABLE [dbo].[Investigator] ADD CONSTRAINT [DF_Investigator_UniqueId] DEFAULT (newid()) FOR [UniqueId]
GO
ALTER TABLE [dbo].[Investigator] ADD CONSTRAINT [DF_Investigator_CreatedTimestamp] DEFAULT (getutcdate()) FOR [CreatedTimestamp]
GO
ALTER TABLE [dbo].[Investigator] ADD CONSTRAINT [DF_Investigator_UpdatedTimestamp] DEFAULT (getutcdate()) FOR [UpdatedTimestamp]
GO
