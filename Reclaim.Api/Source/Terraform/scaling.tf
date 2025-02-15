resource "azurerm_monitor_autoscale_setting" "appsvc" {
  for_each = local.environments
  name                = "AutoscaleAppService${each.key}"
  resource_group_name = azurerm_resource_group.core[each.key].name
  location            = var.location
  target_resource_id  = azurerm_service_plan.api[each.key].id

  profile {
    name = "Scale"

    capacity {
      default = 1
      maximum = 5
      minimum = 1
    }

    rule {
      metric_trigger {
        metric_name = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.api[each.key].id

        time_aggregation = "Average"
        threshold = "90"
        operator = "GreaterThan"
        time_grain = "PT1M"
        time_window = "PT5M"
        statistic = "Average"
      }
      scale_action {
        value = "1"
        type = "ChangeCount"
        direction = "Increase"
        cooldown = "PT1M"
      }
    }
    rule {
      metric_trigger {
        metric_name        = "CpuPercentage"
        metric_resource_id = azurerm_service_plan.api[each.key].id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 25
      }

      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT1M"
      }
    }
    
    rule {
      metric_trigger {
        metric_name = "MemoryPercentage"
        metric_resource_id = azurerm_service_plan.api[each.key].id

        time_aggregation = "Average"
        threshold = "90"
        operator = "GreaterThan"
        time_grain = "PT1M"
        time_window = "PT5M"
        statistic = "Average"
      }
      scale_action {
        value = "1"
        type = "ChangeCount"
        direction = "Increase"
        cooldown = "PT1M"
      }
    }
    rule {
      metric_trigger {
        metric_name        = "MemoryPercentage"
        metric_resource_id = azurerm_service_plan.api[each.key].id
        time_grain         = "PT1M"
        statistic          = "Average"
        time_window        = "PT5M"
        time_aggregation   = "Average"
        operator           = "LessThan"
        threshold          = 25
      }

      scale_action {
        direction = "Decrease"
        type      = "ChangeCount"
        value     = "1"
        cooldown  = "PT1M"
      }
    }
  }
}