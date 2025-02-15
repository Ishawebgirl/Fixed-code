resource "azurerm_resource_group" "core" {
  for_each = local.environments
  name = "reclaim-${each.key}-infra"
  location = var.location
}