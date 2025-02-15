resource "azurerm_static_web_app" "this" {
    for_each = local.environments
    name = "reclaim-${each.key}-swa"
    location = var.location
    resource_group_name = azurerm_resource_group.core[each.key].name    
}

resource "azurerm_dns_cname_record" "this" {
  for_each = local.environments
  name                = "${each.key}"
  zone_name           = "metacast.com"
  resource_group_name = azurerm_resource_group.core["prod"].name
  ttl                 = 300
  record              = azurerm_static_web_app.this[each.key].default_host_name
}

resource "azurerm_static_web_app_custom_domain" "this" {
  for_each = local.environments
  static_web_app_id = azurerm_static_web_app.this[each.key].id
  domain_name       = "${each.key}.metacast.com"
  validation_type   = "cname-delegation"
}

/*
resource "azurerm_app_service_custom_hostname_binding" "swa-hostname" {
  for_each = local.environments
  hostname            = azurerm_dns_zone.this[each.key].zone_name
  app_service_name    = azurerm_static_web_app.this[each.key].name
  resource_group_name = azurerm_static_web_app.this[each.key].resource_group_name
}
*/