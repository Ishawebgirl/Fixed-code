resource "random_password" "db" {
  special = false
  length = 24
}

resource "azurerm_mssql_server" "db" {
  for_each = local.environments
  name                         = "reclaim-${each.key}-mssql"
  resource_group_name          = azurerm_resource_group.core[each.key].name
  location                     = var.location
  version                      = "12.0"
  administrator_login          = "reclaimadmin"
  administrator_login_password = random_password.db.result
  
}

resource "azurerm_mssql_database" "db" {
  for_each = local.environments
  name           = "reclaim-${each.key}-db"
  server_id      = azurerm_mssql_server.db[each.key].id
  collation      = "SQL_Latin1_General_CP1_CI_AS"
  license_type   = "LicenseIncluded"
  max_size_gb    = 1
  sku_name       = "S0"
  zone_redundant = false
  enclave_type   = "VBS"

  lifecycle {
    prevent_destroy = false
  }
}

resource "azurerm_sql_firewall_rule" "db" {
  for_each = local.environments
  name                = "Allow access from Azure resouces"
  resource_group_name = azurerm_resource_group.core[each.key].name
  server_name         = azurerm_mssql_server.db[each.key].name
  start_ip_address    = "0.0.0.0"
  end_ip_address      = "0.0.0.0"
}

output "password" {
  value = random_password.db.result  
  sensitive = true
}