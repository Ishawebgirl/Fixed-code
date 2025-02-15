locals {
  environments = toset(var.environments)
}

resource "azurerm_service_plan" "api" {
  for_each = local.environments
  name = "reclaim-${each.key}-asp"
  location = var.location
  resource_group_name = azurerm_resource_group.core[each.key].name
  sku_name = "B3"
  os_type = "Linux"
  
}

resource "azurerm_linux_web_app" "api" {
  for_each = local.environments
  name = "reclaim-${each.key}-api"
  location = var.location

  resource_group_name = azurerm_resource_group.core[each.key].name
  service_plan_id = azurerm_service_plan.api[each.key].id
  
  app_settings = {
    "RECLAIM_API_CONNECTION_STRING" = "Server=tcp:${azurerm_mssql_server.db[each.key].name}.database.windows.net,1433;Initial Catalog=reclaim-${each.key}-db;Persist Security Info=False;User ID=${azurerm_mssql_server.db[each.key].administrator_login};Password=${azurerm_mssql_server.db[each.key].administrator_login_password};MultipleActiveResultSets=False;Encrypt=True;TrustServerCertificate=False;Connection Timeout=30;"
  }

  site_config {
    application_stack {
      dotnet_version = "8.0"
    }
  }

  logs {
    detailed_error_messages = false
    failed_request_tracing  = false
    http_logs {
        file_system {
            retention_in_days = 7
            retention_in_mb   = 35
            }
        }
    }
}

resource "azurerm_dns_zone" "this" {
  name                = "metacast.com"
  resource_group_name = azurerm_resource_group.core["prod"].name
}

resource "azurerm_dns_txt_record" "domain-verification" {
    for_each = local.environments
    name                = "asuid.api.${each.key}"
    zone_name           = "metacast.com"
    resource_group_name = azurerm_resource_group.core["prod"].name
    ttl                 = 300
    
    record {
        value = azurerm_linux_web_app.api[each.key].custom_domain_verification_id
    }    
}

resource "azurerm_dns_cname_record" "cname-record" {
    for_each = local.environments
    name                = "api.${each.key}"
    zone_name           = azurerm_dns_txt_record.domain-verification[each.key].zone_name
    resource_group_name = azurerm_resource_group.core["prod"].name
    ttl                 = 300
    record              = azurerm_linux_web_app.api[each.key].default_hostname
    
    depends_on = [azurerm_dns_txt_record.domain-verification]
}

resource "azurerm_app_service_custom_hostname_binding" "api-hostname" {
  for_each = local.environments
  hostname            = "api.${each.key}.metacast.com"
  app_service_name    = azurerm_linux_web_app.api[each.key].name
  resource_group_name = azurerm_linux_web_app.api[each.key].resource_group_name
}

resource "azurerm_app_service_managed_certificate" "cert" {
  for_each = local.environments
  custom_hostname_binding_id = azurerm_app_service_custom_hostname_binding.api-hostname[each.key].id
}

resource "azurerm_app_service_certificate_binding" "cert_binding" {
  for_each = local.environments
  hostname_binding_id = azurerm_app_service_custom_hostname_binding.api-hostname[each.key].id
  certificate_id      = azurerm_app_service_managed_certificate.cert[each.key].id
  ssl_state           = "SniEnabled"
}
