ALTER TABLE [dbo].[Policy] WITH CHECK ADD CONSTRAINT [FK_Policy_Customer]
   FOREIGN KEY([CustomerID]) REFERENCES [dbo].[Customer] ([CustomerID])

GO
ALTER TABLE [dbo].[Policy] WITH CHECK ADD CONSTRAINT [FK_Policy_OwnershipType]
   FOREIGN KEY([OwnershipTypeID]) REFERENCES [dbo].[OwnershipType] ([OwnershipTypeID])

GO
ALTER TABLE [dbo].[Policy] WITH CHECK ADD CONSTRAINT [FK_Policy_PropertyType]
   FOREIGN KEY([PropertyTypeID]) REFERENCES [dbo].[PropertyType] ([PropertyTypeID])

GO
ALTER TABLE [dbo].[Policy] WITH CHECK ADD CONSTRAINT [FK_Policy_RoofType]
   FOREIGN KEY([RoofTypeID]) REFERENCES [dbo].[RoofType] ([RoofTypeID])

GO
ALTER TABLE [dbo].[Policy] WITH CHECK ADD CONSTRAINT [FK_Policy_State]
   FOREIGN KEY([StateID]) REFERENCES [dbo].[State] ([StateID])

GO
