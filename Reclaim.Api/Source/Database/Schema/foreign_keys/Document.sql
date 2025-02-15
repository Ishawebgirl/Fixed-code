ALTER TABLE [dbo].[Document] WITH CHECK ADD CONSTRAINT [FK_Document_Account]
   FOREIGN KEY([AccountID]) REFERENCES [dbo].[Account] ([AccountID])

GO
ALTER TABLE [dbo].[Document] WITH CHECK ADD CONSTRAINT [FK_Document_Claim]
   FOREIGN KEY([ClaimID]) REFERENCES [dbo].[Claim] ([ClaimID])

GO
ALTER TABLE [dbo].[Document] WITH CHECK ADD CONSTRAINT [FK_Document_DocumentType]
   FOREIGN KEY([DocumentTypeID]) REFERENCES [dbo].[DocumentType] ([DocumentTypeID])

GO
