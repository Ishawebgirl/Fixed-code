import moment from 'moment'
import { lowerCase, startCase, upperFirst } from 'lodash'

const enums = [
  'type',
  'disposition',
  'status',
  'identityProvider',
  'level',
  'ownershipType',
  'propertyType',
  'role',
  'roofType',
]

const flattenItem = (
  obj: any,
  prefix = '',
  ignoredFields: string[],
  result: any = {}
) => {
  for (let key in obj) {
    const finalKey = startCase(prefix + key)

    if (ignoredFields.includes(key)) {
      continue
    }

    if (enums.includes(key)) {
      result[finalKey] = upperFirst(lowerCase(obj[key]))
      continue
    }

    if (obj[key] instanceof moment) {
      const mo = moment(obj[key])
      const format = mo.creationData().format
      const input = mo.creationData().input?.toString()

      if (input && input.split(':').length === 3 && input.length === 8) {
        result[finalKey] = moment('1/1/2020 ' + input).format('h:mma')
        continue
      }

      switch (format) {
        case 'YYYY-MM-DDTHH:mm:ssZ':
          result[finalKey] = mo.format('MMM DD, YYYY h:mma')
          continue

        case 'YYYY-MM-DD':
          result[finalKey] = mo.format('MMM DD, YYYY')
          continue

        default:
          result[finalKey] = input
          continue
      }
    }

    if (typeof obj[key] === 'object' && obj[key] !== null) {
      Object.assign(result, flattenItem(obj[key], prefix + key + '.', ignoredFields))
    } else {
      result[finalKey] = obj[key]
    }
  }
  return result
}

export const flatten = (obj: any[], ignoredFields: string[]): any[] => {
  const results: any[] = []

  for (const item of obj) {
    const result = flattenItem(item, '', ignoredFields)
    results.push(result)
  }

  return results
}
