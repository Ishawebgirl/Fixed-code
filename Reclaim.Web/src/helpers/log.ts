import moment from 'moment'

const key = 'reclaim_log'

export abstract class Log {
  public static clear = () => {
    localStorage.removeItem(key)
  }

  public static get = () => {
    return localStorage.getItem(key)
  }

  public static add = (entry: string) => {
    const con = console
    con.log(entry)

    let old = localStorage.getItem(key)

    if (old === null) {
      old = ''
    }

    localStorage.setItem(key, `${old}\r\n${moment().format('hh:mma')} ${entry}`)
  }
}
