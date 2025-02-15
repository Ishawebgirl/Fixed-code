ALTER TABLE [dbo].[Investigator] WITH CHECK ADD CONSTRAINT [FK_Investigator_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Investigator] WITH CHECK ADD CONSTRAINT [FK_Investigator_InvestigatorStatus]
   FOREIGN KEY([InvestigatorStatusID]) REFERENCES [dbo].[InvestigatorStatus] ([InvestigatorStatusID])

GO
ALTER TABLE [dbo].[Investigator] WITH CHECK ADD CONSTRAINT [FK_Investigator_State]
   FOREIGN KEY([StateID]) REFERENCES [dbo].[State] ([StateID])

GO
