ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_ClaimDisposition]
   FOREIGN KEY([ClaimDispositionID]) REFERENCES [dbo].[ClaimDisposition] ([ClaimDispositionID])

GO
ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_ClaimStatus]
   FOREIGN KEY([ClaimStatusID]) REFERENCES [dbo].[ClaimStatus] ([ClaimStatusID])

GO
ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_ClaimType]
   FOREIGN KEY([ClaimTypeID]) REFERENCES [dbo].[ClaimType] ([ClaimTypeID])

GO
ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_Investigator]
   FOREIGN KEY([InvestigatorID]) REFERENCES [dbo].[Investigator] ([InvestigatorID])

GO
ALTER TABLE [dbo].[Claim] WITH CHECK ADD CONSTRAINT [FK_Claim_Policy]
   FOREIGN KEY([PolicyID]) REFERENCES [dbo].[Policy] ([PolicyID])

GO
